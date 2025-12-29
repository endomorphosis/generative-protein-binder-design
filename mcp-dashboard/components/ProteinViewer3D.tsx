'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { addToDesignLibrary } from '@/lib/design-library'

interface ProteinViewer3DProps {
  pdbData: string
  onClose: () => void
  title?: string
  sequence?: string
  onUseSequence?: (sequence: string) => void
}

export default function ProteinViewer3D({ pdbData, onClose, title, sequence, onUseSequence }: ProteinViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [renderMode, setRenderMode] = useState<'ribbon' | 'cartoon' | 'sphere' | 'stick'>('ribbon')
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [selectedResidues, setSelectedResidues] = useState<Array<{ chain: string; residueNum: number; residue: string }>>([])
  const [positionsText, setPositionsText] = useState<string>('')
  const [numVariants, setNumVariants] = useState<number>(5)
  const [variantsResult, setVariantsResult] = useState<any>(null)
  const [variantsError, setVariantsError] = useState<string | null>(null)
  const [variantsRunning, setVariantsRunning] = useState(false)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)

  const setObjectMaterialOpacity = (obj: THREE.Object3D, opacity: number, transparent: boolean) => {
    const mesh = obj as THREE.Mesh
    const mat = (mesh as any).material as THREE.Material | THREE.Material[] | undefined
    const applyTo = (m: THREE.Material) => {
      const anyMat = m as any
      if (typeof anyMat.opacity === 'number' && typeof anyMat.transparent === 'boolean') {
        anyMat.opacity = opacity
        anyMat.transparent = transparent
        anyMat.needsUpdate = true
      }
    }

    if (!mat) return
    if (Array.isArray(mat)) {
      mat.forEach(applyTo)
    } else {
      applyTo(mat)
    }
  }

  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Only atoms exist as meshes in Ball & Stick mode; keep other modes untouched.
    if (renderMode !== 'sphere') {
      scene.traverse((obj) => {
        const ud = (obj as any)?.userData
        if (ud?.kind !== 'atom') return
        setObjectMaterialOpacity(obj, 1, false)
      })
      return
    }

    const selectedKeys = new Set(selectedResidues.map((r) => `${r.chain}:${r.residueNum}`))
    const hasSelection = selectedKeys.size > 0

    scene.traverse((obj) => {
      const ud = (obj as any)?.userData
      if (ud?.kind !== 'atom') return

      if (!hasSelection) {
        setObjectMaterialOpacity(obj, 1, false)
        return
      }

      const key = `${String(ud.chain || '')}:${Number(ud.residueNum)}`
      const isSelected = selectedKeys.has(key)

      // Visual emphasis without introducing new colors: selected atoms stay opaque;
      // non-selected atoms become translucent.
      setObjectMaterialOpacity(obj, isSelected ? 1 : 0.25, true)
    })
  }, [renderMode, selectedResidues])

  const parsePositions = (text: string) => {
    const nums = text
      .split(/[^0-9]+/g)
      .map((t) => t.trim())
      .filter(Boolean)
      .map((t) => Number(t))
      .filter((n) => Number.isFinite(n) && n > 0)
    const uniq = Array.from(new Set(nums))
    uniq.sort((a, b) => a - b)
    return uniq
  }

  const callProposeVariants = async () => {
    setVariantsError(null)
    setVariantsResult(null)

    if (!sequence || typeof sequence !== 'string' || !sequence.trim()) {
      setVariantsError('Sequence is not available for this structure')
      return
    }

    const positions = parsePositions(positionsText)
      .filter((p) => p >= 1 && p <= sequence.length)

    if (positions.length === 0) {
      setVariantsError('Select residues (Ball & Stick) or enter positions (1-based)')
      return
    }

    setVariantsRunning(true)
    try {
      const res = await fetch('/api/mcp/tools/call', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'propose_sequence_variants',
          arguments: {
            sequence,
            positions,
            num_variants: numVariants,
          },
        }),
      })

      const payload = await res.json()
      if (!res.ok) {
        throw new Error(payload?.error || `HTTP ${res.status}`)
      }

      const text = payload?.content?.find((c: any) => c?.type === 'text')?.text
      if (typeof text === 'string' && text.trim()) {
        try {
          setVariantsResult(JSON.parse(text))
        } catch {
          setVariantsResult({ variants: [], raw: text })
        }
      } else {
        setVariantsResult(null)
      }
    } catch (e: any) {
      setVariantsError(e?.message || 'Variant proposal failed')
    } finally {
      setVariantsRunning(false)
    }
  }

  useEffect(() => {
    if (!containerRef.current || !pdbData) return

    try {
      // Parse PDB data
      const atoms = parsePDB(pdbData)
      
      if (atoms.length === 0) {
        setError('No valid atoms found in PDB data')
        return
      }

      // Setup scene
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x1a202c)
      sceneRef.current = scene

      // Setup camera
      const camera = new THREE.PerspectiveCamera(
        75,
        containerRef.current.clientWidth / containerRef.current.clientHeight,
        0.1,
        1000
      )
      camera.position.z = 50
      cameraRef.current = camera

      // Setup renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
      containerRef.current.appendChild(renderer.domElement)
      rendererRef.current = renderer

      // Setup controls
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.05
      controls.minDistance = 10
      controls.maxDistance = 200
      controlsRef.current = controls

      // Add lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
      scene.add(ambientLight)

      const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8)
      directionalLight1.position.set(1, 1, 1)
      scene.add(directionalLight1)

      const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4)
      directionalLight2.position.set(-1, -1, -1)
      scene.add(directionalLight2)

      // Create molecule
      createMolecule(scene, atoms, renderMode, showHeatmap)

      // Center camera on molecule
      const box = new THREE.Box3().setFromObject(scene)
      const center = box.getCenter(new THREE.Vector3())
      const size = box.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z)
      const fov = camera.fov * (Math.PI / 180)
      let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2))
      cameraZ *= 1.5
      camera.position.set(center.x, center.y, center.z + cameraZ)
      camera.lookAt(center)
      controls.target.copy(center)

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate)
        controls.update()
        renderer.render(scene, camera)
      }
      animate()

      const raycaster = new THREE.Raycaster()
      const mouse = new THREE.Vector2()
      const handleClick = (evt: MouseEvent) => {
        if (renderMode !== 'sphere') return
        if (!rendererRef.current || !cameraRef.current || !sceneRef.current) return

        const rect = renderer.domElement.getBoundingClientRect()
        mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1
        mouse.y = -(((evt.clientY - rect.top) / rect.height) * 2 - 1)
        raycaster.setFromCamera(mouse, cameraRef.current)

        const intersects = raycaster.intersectObjects(sceneRef.current.children, true)
        const hit = intersects.find((i) => (i.object as any)?.userData?.kind === 'atom')
        const ud = (hit?.object as any)?.userData
        if (!ud) return

        const chain = String(ud.chain || '')
        const residueNum = Number(ud.residueNum)
        const residue = String(ud.residue || '')
        if (!Number.isFinite(residueNum)) return

        setSelectedResidues((prev) => {
          const exists = prev.some((r) => r.chain === chain && r.residueNum === residueNum)
          const next = exists
            ? prev.filter((r) => !(r.chain === chain && r.residueNum === residueNum))
            : [...prev, { chain, residueNum, residue }]

          const pos = Array.from(new Set(next.map((r) => r.residueNum))).sort((a, b) => a - b)
          setPositionsText(pos.join(','))
          return next
        })
      }
      renderer.domElement.addEventListener('click', handleClick)

      // Handle resize
      const handleResize = () => {
        if (!containerRef.current || !camera || !renderer) return
        camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
      }
      window.addEventListener('resize', handleResize)

      return () => {
        window.removeEventListener('resize', handleResize)
        renderer.domElement.removeEventListener('click', handleClick)
        if (containerRef.current && renderer.domElement) {
          containerRef.current.removeChild(renderer.domElement)
        }
        renderer.dispose()
        controls.dispose()
      }
    } catch (err) {
      setError(`Failed to render 3D structure: ${err}`)
    }
  }, [pdbData, renderMode, showHeatmap])

  const parsePDB = (pdb: string) => {
    const atoms: Array<{
      element: string
      x: number
      y: number
      z: number
      residue: string
      residueNum: number
      chain: string
      atomName: string
      bFactor: number
    }> = []

    const lines = pdb.split('\n')
    for (const line of lines) {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        try {
          const atomName = line.substring(12, 16).trim()
          const element = line.substring(76, 78).trim() || atomName.charAt(0)
          const x = parseFloat(line.substring(30, 38))
          const y = parseFloat(line.substring(38, 46))
          const z = parseFloat(line.substring(46, 54))
          const residue = line.substring(17, 20).trim()
          const residueNum = parseInt(line.substring(22, 26).trim())
          const chain = line.substring(21, 22).trim()
          const bFactor = parseFloat(line.substring(60, 66).trim()) || 0

          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            atoms.push({ element, x, y, z, residue, residueNum, chain, atomName, bFactor })
          }
        } catch (err) {
          // Skip malformed lines
        }
      }
    }

    return atoms
  }

  const detectSecondaryStructure = (
    caAtoms: Array<{ x: number; y: number; z: number; residueNum: number }>
  ): Array<{ start: number; end: number; type: 'helix' | 'sheet' | 'turn' | 'coil' }> => {
    const structures: Array<{ start: number; end: number; type: 'helix' | 'sheet' | 'turn' | 'coil' }> = []
    
    // Simple heuristic: detect helices and sheets based on C-alpha distances and angles
    for (let i = 0; i < caAtoms.length - 3; i++) {
      const dist1 = Math.sqrt(
        Math.pow(caAtoms[i].x - caAtoms[i + 3].x, 2) +
        Math.pow(caAtoms[i].y - caAtoms[i + 3].y, 2) +
        Math.pow(caAtoms[i].z - caAtoms[i + 3].z, 2)
      )
      
      // Helix: i to i+3 distance ~5.4Å
      if (dist1 >= 4.5 && dist1 <= 6.5) {
        let helixEnd = i + 3
        while (helixEnd < caAtoms.length - 1) {
          const nextDist = Math.sqrt(
            Math.pow(caAtoms[helixEnd - 3].x - caAtoms[helixEnd].x, 2) +
            Math.pow(caAtoms[helixEnd - 3].y - caAtoms[helixEnd].y, 2) +
            Math.pow(caAtoms[helixEnd - 3].z - caAtoms[helixEnd].z, 2)
          )
          if (nextDist < 4.5 || nextDist > 6.5) break
          helixEnd++
        }
        if (helixEnd - i >= 4) {
          structures.push({ start: i, end: helixEnd, type: 'helix' })
        }
      }
    }
    
    return structures
  }

  const createMolecule = (
    scene: THREE.Scene,
    atoms: Array<{ element: string; x: number; y: number; z: number; residue: string; residueNum: number; chain: string; atomName: string; bFactor: number }>,
    mode: 'ribbon' | 'cartoon' | 'sphere' | 'stick',
    heatmap: boolean
  ) => {
    // Clear existing molecule
    scene.children = scene.children.filter(
      child => !(child instanceof THREE.Mesh || child instanceof THREE.Line)
    )

    const colors: Record<string, number> = {
      C: 0x909090,
      N: 0x3050f8,
      O: 0xff0d0d,
      S: 0xffff30,
      H: 0xffffff,
      P: 0xff8000,
      default: 0xff00ff
    }

    const secondaryStructureColors = {
      helix: 0xff4081,    // Pink/magenta for alpha helices
      sheet: 0xffd600,    // Yellow for beta sheets
      turn: 0x00bcd4,     // Cyan for turns
      coil: 0x9e9e9e      // Gray for random coil
    }

    if (mode === 'ribbon' || mode === 'cartoon') {
      // Get C-alpha atoms for ribbon/cartoon representation
      const caAtoms = atoms.filter(a => a.atomName === 'CA')
      
      if (caAtoms.length === 0) {
        // Fallback if no CA atoms found
        setError('No C-alpha atoms found for ribbon visualization')
        return
      }

      // Detect secondary structure
      const structures = detectSecondaryStructure(caAtoms)

      // Create ribbon with arrows for directionality
      for (let i = 0; i < caAtoms.length - 1; i++) {
        const curr = caAtoms[i]
        const next = caAtoms[i + 1]
        
        // Determine secondary structure type for this segment
        let structureType: 'helix' | 'sheet' | 'turn' | 'coil' = 'coil'
        for (const struct of structures) {
          if (i >= struct.start && i < struct.end) {
            structureType = struct.type
            break
          }
        }

        const color = heatmap 
          ? new THREE.Color().setHSL((1 - curr.bFactor / 100) * 0.7, 1, 0.5)
          : new THREE.Color(secondaryStructureColors[structureType])

        // Create ribbon segment
        const direction = new THREE.Vector3(
          next.x - curr.x,
          next.y - curr.y,
          next.z - curr.z
        )
        const length = direction.length()
        direction.normalize()

        // For helices: use wider ribbon with twist
        // For sheets: use flat arrow ribbons
        // For coils: use thin tube

        if (structureType === 'helix' && mode === 'ribbon') {
          // Helical ribbon - coil shape
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(curr.x, curr.y, curr.z),
            new THREE.Vector3(next.x, next.y, next.z)
          ])
          const tubeGeometry = new THREE.TubeGeometry(curve, 8, 0.4, 8, false)
          const material = new THREE.MeshPhongMaterial({ 
            color, 
            side: THREE.DoubleSide,
            shininess: 30
          })
          const tube = new THREE.Mesh(tubeGeometry, material)
          scene.add(tube)
        } else if (structureType === 'sheet' && mode === 'ribbon') {
          // Beta sheet - flat arrow ribbon showing directionality
          const arrowWidth = 0.8
          const arrowHeight = 0.1
          
          // Create arrow shape pointing in direction of backbone
          const shape = new THREE.Shape()
          shape.moveTo(-arrowWidth/2, 0)
          shape.lineTo(arrowWidth/2, 0)
          shape.lineTo(arrowWidth/2, length * 0.7)
          shape.lineTo(arrowWidth * 0.7, length * 0.7)
          shape.lineTo(0, length)
          shape.lineTo(-arrowWidth * 0.7, length * 0.7)
          shape.lineTo(-arrowWidth/2, length * 0.7)
          shape.lineTo(-arrowWidth/2, 0)

          const extrudeSettings = {
            steps: 1,
            depth: arrowHeight,
            bevelEnabled: false
          }

          const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings)
          const material = new THREE.MeshPhongMaterial({ 
            color,
            side: THREE.DoubleSide,
            shininess: 50
          })
          const arrow = new THREE.Mesh(geometry, material)
          
          // Position and orient the arrow
          arrow.position.set(curr.x, curr.y, curr.z)
          arrow.lookAt(next.x, next.y, next.z)
          arrow.rotateX(Math.PI / 2)
          
          scene.add(arrow)
        } else {
          // Coil/turn - simple smooth tube
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(curr.x, curr.y, curr.z),
            new THREE.Vector3(next.x, next.y, next.z)
          ])
          const tubeGeometry = new THREE.TubeGeometry(curve, 4, 0.2, 6, false)
          const material = new THREE.MeshPhongMaterial({ color })
          const tube = new THREE.Mesh(tubeGeometry, material)
          scene.add(tube)
        }
      }
    } else if (mode === 'sphere') {
      // Ball-and-stick representation
      atoms.forEach(atom => {
        const geometry = new THREE.SphereGeometry(0.3, 16, 16)
        const color = colors[atom.element] || colors.default
        const material = new THREE.MeshPhongMaterial({ color })
        const sphere = new THREE.Mesh(geometry, material)
        ;(sphere as any).userData = {
          kind: 'atom',
          chain: atom.chain,
          residueNum: atom.residueNum,
          residue: atom.residue,
          atomName: atom.atomName,
        }
        sphere.position.set(atom.x, atom.y, atom.z)
        scene.add(sphere)
      })

      // Add bonds (simple distance-based)
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const dist = Math.sqrt(
            Math.pow(atoms[i].x - atoms[j].x, 2) +
            Math.pow(atoms[i].y - atoms[j].y, 2) +
            Math.pow(atoms[i].z - atoms[j].z, 2)
          )
          if (dist < 1.8) {
            const points = [
              new THREE.Vector3(atoms[i].x, atoms[i].y, atoms[i].z),
              new THREE.Vector3(atoms[j].x, atoms[j].y, atoms[j].z)
            ]
            const geometry = new THREE.BufferGeometry().setFromPoints(points)
            const material = new THREE.LineBasicMaterial({ color: 0x666666 })
            const line = new THREE.Line(geometry, material)
            scene.add(line)
          }
        }
      }
    } else if (mode === 'stick') {
      // Stick representation
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const dist = Math.sqrt(
            Math.pow(atoms[i].x - atoms[j].x, 2) +
            Math.pow(atoms[i].y - atoms[j].y, 2) +
            Math.pow(atoms[i].z - atoms[j].z, 2)
          )
          if (dist < 1.8) {
            const points = [
              new THREE.Vector3(atoms[i].x, atoms[i].y, atoms[i].z),
              new THREE.Vector3(atoms[j].x, atoms[j].y, atoms[j].z)
            ]
            const geometry = new THREE.BufferGeometry().setFromPoints(points)
            const color = colors[atoms[i].element] || colors.default
            const material = new THREE.LineBasicMaterial({ color, linewidth: 3 })
            const line = new THREE.Line(geometry, material)
            scene.add(line)
          }
        }
      }
    }
  }

  const changeRenderMode = (mode: 'ribbon' | 'cartoon' | 'sphere' | 'stick') => {
    setRenderMode(mode)
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-6xl h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              🔬 3D Protein Structure Viewer
            </h3>
            {title && <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{title}</p>}
          </div>
          <button
            onClick={onClose}
            aria-label="Close 3D Viewer"
            data-testid="close-3d-viewer"
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 p-2"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex gap-2 items-center flex-wrap">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Render Mode:</span>
          <button
            onClick={() => changeRenderMode('ribbon')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'ribbon'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            🎀 Ribbon (Biochem)
          </button>
          <button
            onClick={() => changeRenderMode('sphere')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'sphere'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Ball & Stick
          </button>
          <button
            onClick={() => changeRenderMode('stick')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'stick'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Stick
          </button>
          <button
            onClick={() => changeRenderMode('cartoon')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'cartoon'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Cartoon
          </button>
          
          <div className="border-l border-gray-300 dark:border-gray-600 h-8 mx-2"></div>
          
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              showHeatmap
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
            title="B-factor heatmap (thermal motion/flexibility)"
          >
            🔥 B-Factor Heatmap
          </button>

          {sequence && (
            <>
              <div className="border-l border-gray-300 dark:border-gray-600 h-8 mx-2"></div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Variants:</span>
                <input
                  aria-label="Variant positions"
                  data-testid="variant-positions"
                  value={positionsText}
                  onChange={(e) => setPositionsText(e.target.value)}
                  placeholder="positions (1-based) e.g. 12,15,16"
                  className="px-3 py-2 rounded text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 w-64"
                />
                <input
                  aria-label="Number of variants"
                  data-testid="variant-num"
                  type="number"
                  min={1}
                  max={20}
                  value={numVariants}
                  onChange={(e) => {
                    const n = Number(e.target.value)
                    setNumVariants(Number.isFinite(n) && n >= 1 ? Math.min(20, Math.floor(n)) : 5)
                  }}
                  className="px-3 py-2 rounded text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 w-24"
                />
                <button
                  onClick={callProposeVariants}
                  data-testid="propose-variants"
                  disabled={variantsRunning}
                  className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                    variantsRunning
                      ? 'bg-gray-300 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                  title={renderMode === 'sphere' ? 'Click atoms to select residues' : 'Switch to Ball & Stick to select residues'}
                >
                  {variantsRunning ? 'Proposing…' : 'Propose Variants'}
                </button>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {renderMode === 'sphere'
                    ? `Selected: ${selectedResidues.length} residue(s). Click atoms to toggle.`
                    : `Selected: ${selectedResidues.length} residue(s). Use Ball & Stick to click.`}
                </span>
              </div>
            </>
          )}
          
          <div className="ml-auto text-xs text-gray-500 dark:text-gray-400">
            💡 Use mouse to rotate, scroll to zoom
          </div>
        </div>

        {/* 3D Viewer */}
        <div className="flex-1 relative">
          {error ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-6xl mb-4">⚠️</div>
                <p className="text-red-600 dark:text-red-400">{error}</p>
              </div>
            </div>
          ) : (
            <div ref={containerRef} className="w-full h-full" />
          )}
        </div>

        {/* Footer with legend */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          {(variantsError || (variantsResult && variantsResult?.variants?.length)) && (
            <div className="mb-3">
              {variantsError && (
                <div className="text-sm text-red-600 dark:text-red-400">{variantsError}</div>
              )}
              {variantsResult?.variants?.length ? (
                <div className="space-y-2">
                  <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Proposed Variants (best-first)
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {variantsResult.variants.slice(0, 6).map((v: any, idx: number) => (
                      <div
                        key={idx}
                        className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded p-2"
                      >
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium text-gray-800 dark:text-gray-200">Score</span>
                          <span
                            data-testid={`variant-score-${idx}`}
                            className="text-gray-700 dark:text-gray-300"
                          >
                            {String(v?.score ?? '')}
                          </span>
                        </div>
                        {typeof v?.sequence === 'string' && onUseSequence && (
                          <div className="mt-2">
                            <button
                              data-testid={`iterate-variant-${idx}`}
                              onClick={() => {
                                onUseSequence(v.sequence)
                                onClose()
                              }}
                              className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2 px-3 rounded transition-colors"
                            >
                              Iterate with this
                            </button>
                          </div>
                        )}
                        {typeof v?.sequence === 'string' && (
                          <div className="mt-2">
                            <button
                              data-testid={`save-variant-${idx}`}
                              onClick={() => {
                                addToDesignLibrary({
                                  sequence: v.sequence,
                                  score: typeof v?.score === 'number' ? v.score : undefined,
                                  positions: Array.isArray(v?.positions) ? v.positions : undefined,
                                  source: title || '3D Viewer Variant',
                                  pdbData,
                                })
                              }}
                              className="w-full bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium py-2 px-3 rounded transition-colors"
                            >
                              Save to Library
                            </button>
                          </div>
                        )}
                        <div
                          data-testid={`variant-sequence-${idx}`}
                          className="mt-1 font-mono text-xs text-gray-700 dark:text-gray-300 break-all"
                        >
                          {String(v?.sequence ?? '')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}
          <div className="flex items-center gap-6 text-sm flex-wrap">
            {renderMode === 'ribbon' || renderMode === 'cartoon' ? (
              <>
                <span className="font-medium text-gray-700 dark:text-gray-300">Secondary Structure:</span>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{background: '#ff4081'}}></div>
                  <span className="text-gray-600 dark:text-gray-400">α-Helix</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{background: '#ffd600'}}></div>
                  <span className="text-gray-600 dark:text-gray-400">β-Sheet</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{background: '#00bcd4'}}></div>
                  <span className="text-gray-600 dark:text-gray-400">β-Turn</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-gray-400"></div>
                  <span className="text-gray-600 dark:text-gray-400">Random Coil</span>
                </div>
              </>
            ) : (
              <>
                <span className="font-medium text-gray-700 dark:text-gray-300">Element Colors:</span>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-gray-500"></div>
                  <span className="text-gray-600 dark:text-gray-400">Carbon</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                  <span className="text-gray-600 dark:text-gray-400">Nitrogen</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-red-500"></div>
                  <span className="text-gray-600 dark:text-gray-400">Oxygen</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-yellow-400"></div>
                  <span className="text-gray-600 dark:text-gray-400">Sulfur</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-orange-500"></div>
                  <span className="text-gray-600 dark:text-gray-400">Phosphorus</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
